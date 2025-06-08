# 根据geojson处理svs图片
# 分割成slice图片和normalized_slice图片

from os.path import join
from utils.apply_mask import *
from utils.util import *
from cfg import parse_args
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

args=parse_args()

def process_single(num):
    svs_path=join(args.data_dir,f'{num}.svs')
    geojson_path=join(args.data_dir,f'{num}.geojson')
    regions=apply_mask(svs_path,geojson_path)
    seperate_to_slice_image(args,num,regions)
    print(f"Processed {num}")

nums=args.nums
def main():
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_single, num) for num in nums]
        
        for future in tqdm(as_completed(futures), total=len(nums)):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error processing: {e}")

if __name__ == "__main__":
    main()