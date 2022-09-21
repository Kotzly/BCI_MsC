import numpy as np

def block_summing(A):
    K, M, M = A.shape
    R = np.zeros((K * M, K * M))
    for k in range(K):
        s = slice(k * M, (k + 1) * M)
        R[s, s] = A[k]
    return R
def IVA_GN(Z, lr=1e-2, epochs=100, A=None):
    K, M, N = Z.shape
    delta_history = np.empty((K, M, epochs))
    ISI_history = np.empty((K, epochs))
    ISR_history = np.empty((K, epochs))
    W = np.stack([np.eye(M) for k in range(K)])

    for i in tqdm(range(epochs)):
        
        if A is not None:
            for k in range(K):
                ISI_history[k, i] = (ISI(W[[k]], A[[k]]))
                ISR_history[k, i] = ISR(
                    W[[k]],
                    A[[k]]
                )
        Y = np.stack([W[k] @ Z[k] for k in range(K)])
        H = np.stack([recursive_h(W[k]) for k in range(K)]) # K x M x M x 1

        sigma_inv = np.empty((M, K, K))

        for n in range(M):
            sigma_inv[n] = np.linalg.inv(Y[:, n] @ Y[:, n].T / N)
    
        for n in range(M):
            deltas = list() # M x K
            for k in range(K):
                e = np.zeros((K, 1))
                e[k] = 1
                
                delta = (Z[k] @ Y[:, n].T / N) @ sigma_inv[n] @ e
                delta -= H[k, n] / (H[k, n].T @ W[k, n])
                delta = delta.flatten()
                deltas.append(delta)
            deltas = np.stack(deltas)
            
            sigma_inv[n] = np.linalg.inv(Y[:, n] @ Y[:, n].T / N)
    
            B = np.stack(
                [
                    H[k, n] @ H[k, n].T / (H[k, n].T @ W[k, n]) ** 2
                    for k in range(K)
                ]
            )
            BB = block_summing(B)
            X = Z.reshape(K * M, N)
            Rx = X @ X.T / N
#             print(sigma_inv[n].shape, Z.shape)#(Z @ Z.T / N).shape)
            Hscv = np.kron(sigma_inv[n], Rx)[:K * M, :K * M] + BB
    
            deltas = deltas.reshape(-1, 1)
            delta = np.linalg.inv(Hscv) @ deltas
#             print(W[k, n].shape, delta[k * M:(k + 1) * M].shape)
            for k in range(K):
                W[k, n] -= lr * delta[k * M:(k + 1) * M].flatten()
                W[k, n] /= norm(W[k, n])
    return W
        