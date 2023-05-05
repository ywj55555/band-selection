# band-selection
这个项目是我毕设波段选择部分的实验代码，包括以下几个部分：  
1、基于深度学习的嵌入式波段选择模型，bs_embdding.png为模型结构图，BS_embedding_add_water.py为训练代码，BSNET_Conv.py为模型定义文件  
2、复现了现有模型如下：  
  BSNETs:BStrainShAndWater.py为训练代码，模型在BSNET_Conv.py；  
  精英蚁群优化：Aco_BandSelect.py，一种智能算法  
  SFS\SBS：search_band_selection_based_JM.py，二者为贪心算法  
  使用方法：
  1、python main.py --dataset Indian_pines_corrected --method SRL-SOA --weights False --q 3 --bands 25  
  2、python -u BStrainKSC.py > output_ksc_bsnets.log 2>&1 &    
  3、interBandRedundancy.py注意修改utilsF.py 中的 get_class_distributionIP的 count_dict 为相应数据集类别数，或者自己新建一个函数  
3、使用了四个波段选择评价指标：  
  oif、cal_mean_spectral_divergence、get_average_jm_score、get_average_spectral_angle_score，均在bsUtils.py  
  还计算了一个归一化分数，具体见calc_all_index_score.py  
    
注：上海数据集为31类别，衣物和其他有多个类别，把31类划分为4类别，然后借鉴笛卡尔积思想，计算各类之间的JM距离或者spectral_angle_score，然后取平均值，这样较为合理。  
具体划分规则如下：  
cloth = [1, 2, 3, 5, 6, 7]  
other = [4]  
other.extend([x for x in range(9, 30)])  
skin = 8  
我还定义了water = 0  

  
