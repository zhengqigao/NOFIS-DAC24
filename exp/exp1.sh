
cd ..

mkdir ./results/case101/
python main_nofis.py --gpu 0 --testcase 101 --plt_fig save --save_data 1 --save_path ./results/case101/

mkdir ./results/case102/
python main_nofis.py --gpu 0 --testcase 102 --plt_fig save --save_data 1 --save_path ./results/case102/

mkdir ./results/case103/
python main_nofis.py --gpu 0 --testcase 103 --plt_fig save --save_data 1 --save_path ./results/case103/

mkdir ./results/case105/
python main_nofis.py --gpu 0 --testcase 105 --plt_fig save --save_data 1 --save_path ./results/case105/

mkdir ./results/case106/
python main_nofis.py --gpu 0 --testcase 106 --plt_fig save --save_data 1 --save_path ./results/case106/