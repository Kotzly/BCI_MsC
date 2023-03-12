echo 'After running this script, go to notebooks/OpenBMI_Structuring.ipynb'
cat openbmi_urls.txt | xargs -n 1 -P 8 wget
