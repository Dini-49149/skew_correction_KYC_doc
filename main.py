import cv2
from src.skew_correction import skew_correction_passport
from src.image_utils import imShow
import argparse



def main():
    parser = argparse.ArgumentParser(description='Skew correction of KYC documents arguments for KYC document (passport).')
    parser.add_argument('--img_path', type=str, help='The path to the passport image file.')
    
    args = parser.parse_args()
    img_path = args.img_path

    gray, image_rlsa_horizontal_inv, res = skew_correction_passport(img_path)
    cv2.imwrite('data/processed/processed.jpg', gray)
    cv2.imwrite('data/processed/rlsa_horizontal.jpg', image_rlsa_horizontal_inv)
    cv2.imwrite('results/result.jpg', res)
    imShow(res)

if __name__ == "__main__":
    main()
