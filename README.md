# ThermalDetection

# Prerequisites
You will need libusb and CMake installed.
<pre><code>git clone https://github.com/libuvc/libuvc</code></pre>
<pre><code>cd libuvc
mkdir build
cd build
cmake ..
make && sudo make install</code></pre>

# Run the project
1. Download project.
<pre><code>git clone https://github.com/weiweiweiting/ThermalDetection.git</code></pre>

2. Set the environment and run.
Go to project directory.
<pre><code>cd ThermalDetection/</code></pre>
Build a virtual environment.
<pre><code>virtualenv env</code></pre>
Run the environment. **Run this line everytime if you run the project.*
<pre><code>source env/bin/activate</code></pre>
Install package.
<pre><code>pip install opencv-python
pip install PyQt5
pip install pillow
pip install tensorflow
pip install pyzbar
pip install imutils
pip install dlib</code></pre>

3. Run the project
<pre><code>python FaceandQRC9.py</code></pre>

# Modify config file
1. Open <code>config.ini</code>
2. Comment out device_direction = vertical if you want to have horizontal version.
   <br>Comment out device_direction = horizontal if you want to have vertical version.
3. Modify other settings if you need.
