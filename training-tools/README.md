# lczero-training-tools
Some tools to assist you in getting data for Leela Chess Zero

# HOW TO USE
## Prepare your data
Prepare your data by going to [download site](https://storage.lczero.org/files/training_data/test80/) \
Copy as many lines as you want. When your done, put them in linksdirty.txt (see linksdirty.txt for an example) \
Run `python3 cleanlinks.py` to generate links.txt (see links.txt for an example)\
Create a folder called lctd (Leela Chess Training Data)

## Download the data
Download your data by running `python3 getdata.py` \
A progress bar will show your downloads. 10 files will be downloaded at a time. \
After your files finish being downloaded, run `python3 untar.py ./lctd/` \
Again, you have a progress bar. One shows the overall progress for the extraction, and the other shows the progress for each file.

## That's all!
Hopefully you find these tools helpful in your chess endevours. Good luck and happy training!

### Note: Windows users may need to change the paths as this code is for linux. It should work on Windows but I am unable to test.
