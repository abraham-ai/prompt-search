sudo add-apt-repository universe
sudo apt-get install -y curl
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
python3 get-pip.py
pip install autofaiss==2.15.3 clip-by-openai==1.1 onnxruntime==1.12.1 onnx==1.11.0 python-dotenv
pip install git+https://github.com/Lednik7/CLIP-ONNX.git --no-deps
pip install git+https://github.com/abraham-ai/eden
pip install --upgrade protobuf==3.20.0