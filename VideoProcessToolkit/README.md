Video Process Toolkit
====

Video Process Toolkit provide some useful python script to process video.

Scripts
------------------
- **video_tool.py**: main entrance for video process tools, which contain: video cut,crop,format convert,downsample,resize,append,insert,convert to images.
- **video2img.py**: convert video to a list of images.
- **video_combine.py**: combine two same-size videos, append the second video to the end of the first video
- **video_compose.py**: compose two videos in to one video, insert the second video as a sub-window into the first video.
- **video_convert.py**: convert video file format between: avi,ogv,mp4...
- **video_cut.py**: a ui tool to cut video or select sub-window video(crop video).
- **video_downsample.py**: downsample video by skipping frames.
- **video_resize.py**: resize the image size of video.


How to use
-------------------
```python
    video_tool.py input_video [-c | -v | -d DOWNSAMPLE | -r RESIZE | -a APPEND | -i INSERT | -t TOIMAGES] [-o OUTPUT]
    # cut/crop video
    video_tool.py input.avi -c
    # convert video format
    video_tool.py input.avi -v -o ouput.mp4
    # downsample to half frame count
    video_tool.py input.avi -d 2 -o output.avi
    # resize to half image size
    video_tool.py input.avi -r 0.5 -o output.avi
    # append video
    video_tool.py video1.avi -a video2.avi -o output.avi
    # insert video
    video_tool.py video1.avi -i video2.avi -o output.avi
    # convert video to images
    video_tool.py video.avi -t image_save_dir
```