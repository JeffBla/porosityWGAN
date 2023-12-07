dataDirHead="data/rockXCT_fractional_pred_resnet_noPreProcess2/"

dataDirAT_1_3="AT-1_3(1~98.5m)"
dataDirAT_1_15="AT-1_15(2~99.4m_export-2)"
dataDirAT_1_20="AT-1_20(1.1~98.6m)"
dataDirAT_1_69="AT-1_69(1.3~98.8m)"
dataDirAT_1_76="AT-1_76(1.2~98.2m_export-2)"
dataDirAT_1_86="AT-1_86(1.7~98.1m)"
dataDirAT_1_92="AT-1_92(1.6~98.6m_export-2)"
dataDirAT_1_99="AT-1_99(1.4~98.9m_export-2)"
dataDirBH_3_3="BH-3_3(1~98.8m)"
dataDirBH_3_15="BH-3_15(1~98.8m_export-2)"
dataDirBH_3_19="BH-3_19(1~98.6m)"

outDirHead="percentOutput/resnet_pred/"
outputPrefix="resnet18_smallDataset_noPreProcess_"

dataDirArr_train=($dataDirAT_1_3 $dataDirAT_1_15 $dataDirAT_1_20 $dataDirAT_1_69
    $dataDirAT_1_76 $dataDirAT_1_86 $dataDirAT_1_92 $dataDirAT_1_99
    $dataDirBH_3_3 $dataDirBH_3_15 $dataDirBH_3_19)

# put the result together -> train and test
for dataDir in ${dataDirArr_train[@]}; do
    python normal_DL_Model.py \
        "--dataroot_pred" \
        "$dataDirHead$dataDir" \
        "--csv_file_pred_output" \
        "$outDirHead$outputPrefix$dataDir".csv

done
