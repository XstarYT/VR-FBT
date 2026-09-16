"""Install the optional, checksum-pinned DWPose model without changing packages."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import os
import tempfile
import urllib.request
import zipfile
from Lib.Refinement import MODEL_PATH, MODEL_SHA256, MODEL_URL, verify_model


def install():
    if MODEL_PATH.is_file() and hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256:
        print(verify_model(MODEL_PATH));return
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dwpose-',dir=MODEL_PATH.parent) as temporary:
        archive=Path(temporary)/'model.zip'
        print('Downloading optional DWPose model from OpenMMLab...',flush=True)
        with urllib.request.urlopen(MODEL_URL,timeout=60) as response,archive.open('wb') as output:
            total=0
            while chunk:=response.read(1024*1024):
                total+=len(chunk)
                if total>200*1024*1024:raise ValueError('Unexpected model archive size')
                output.write(chunk)
        with zipfile.ZipFile(archive) as bundle:
            members=[item for item in bundle.infolist() if item.filename.endswith('/end2end.onnx')]
            if len(members)!=1 or members[0].file_size>200*1024*1024:raise ValueError('Unexpected model archive contents')
            data=bundle.read(members[0])
        if hashlib.sha256(data).hexdigest()!=MODEL_SHA256:raise ValueError('Downloaded model checksum mismatch')
        staged=Path(temporary)/'model.onnx';staged.write_bytes(data)
        os.replace(staged,MODEL_PATH)
    print(verify_model(MODEL_PATH))

if __name__=='__main__':install()
