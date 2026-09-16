"""Create a measured lens JSON from native, unrotated checkerboard photographs."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,help='Folder of full-resolution JPG/PNG frames from the tracking video mode')
    parser.add_argument('--output',type=Path,help='New JSON file; import using Camera layout > Import lens')
    parser.add_argument('--board',type=Path,help='Write a printable A4 landscape 25 mm checkerboard SVG')
    args=parser.parse_args()
    if args.board:
        squares=''.join(f'<rect x="{23.5+x*25}" y="{17.5+y*25}" width="25" height="25"/>' for y in range(7) for x in range(10) if (x+y)%2==0)
        text='<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 297 210"><rect width="297" height="210" fill="white"/>'+squares+'</svg>'
        with args.board.open('x',encoding='utf-8') as file:file.write(text)
        print(f'Print at 100% scale, landscape, no fit-to-page: {args.board}')
    if not args.images:
        if args.board:return
        parser.error('--images and --output are required for calibration')
    if args.output is None:parser.error('--output is required')
    if args.output.exists():parser.error('Choose a new output filename; existing calibration will not be overwritten')
    import cv2
    from Lib.Calibration import calibrate_lens
    size=None;corners=[]
    files=sorted(path for path in args.images.iterdir() if path.suffix.lower() in {'.jpg','.jpeg','.png'})
    if len(files)>200:parser.error('Use at most 200 selected frames')
    for path in files:
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None:raise ValueError(f'Cannot read {path}')
        current=(image.shape[1],image.shape[0])
        if size is not None and current!=size:raise ValueError('Every calibration image must use the same video resolution')
        size=current
        found,points=cv2.findChessboardCornersSB(image,(9,6),flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:corners.append(points)
        print(f'{path.name}: {"board found" if found else "skipped - no board"}',flush=True)
    report=calibrate_lens(corners,size)
    with args.output.open('x',encoding='utf-8') as file:json.dump(report,file,indent=2)
    print(f'Validated {len(corners)} views. Import {args.output} in Camera layout.')


if __name__=='__main__':main()
