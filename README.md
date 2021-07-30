# ThermalDetection

# Run the project
1. Download project.
<pre><code>git clone https://github.com/weiweiweiting/ThermalDetection.git</code></pre>

2. Set the environment.

Go to project directory.
<pre><code>cd ThermalDetection/</code></pre>
Build a virtual environment.
<pre><code>virtualenv env</code></pre>
Run the environment. **Run this line everytime if you run the project.*
<pre><code>source env/bin/activate</code></pre>
Install package. **PyQt5 may takes some time to install.*
<pre><code>pip install opencv-python
pip install PyQt4
pip install pillow
pip install tensorflow
pip install pyzbar
pip install imutils
pip install dlib</code></pre>

3. Run the environment. 
<pre><code>cd ThermalDetection/</code></pre>
<pre><code>source env/bin/activate</code></pre>

4. Run the project
<pre><code>python FaceandQRC9.py</code></pre>

# Modify config file
1. Open <code>config.ini</code>
2. Comment out device_direction = vertical if you want to have horizontal version.
   <br>Comment out device_direction = horizontal if you want to have vertical version.
3. Modify other settings if you need.
