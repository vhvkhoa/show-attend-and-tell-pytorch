TRAINYEAR=2017
VALYEAR=2017
TESTYEAR=2014

TRAIN=1
VAL=1
TEST=1

if [! $TRAINYEAR -eq $VALYEAR] && [$TRAIN -eq 1]
then
    wget http://images.cocodataset.org/annotations/annotations_trainval$TRAINYEAR.zip -P data/
    unzip data/annotations_trainval$TRAINYEAR.zip -d data/
    rm annotations_trainval$TRAINYEAR.zip
fi

if [$TRAIN -eq 1]
then
    wget http://images.cocodataset.org/zips/train$TRAINYEAR.zip -P image/
    unzip image/train$TRAINYEAR.zip -d image/ && mv image/train$TRAINYEAR image/train
    rm image/train$TRAINYEAR.zip
    rm data/annotations_trainval$VALYEAR.zip
fi

if [$VAL -eq 1]
then
    wget http://images.cocodataset.org/annotations/annotations_trainval$VALYEAR.zip -P data/
    wget http://images.cocodataset.org/zips/val$VALYEAR.zip -P image/
    unzip data/annotations_trainval$VALYEAR.zip -d data/
    unzip image/val$VALYEAR.zip -d image/ && mv image/val$VALYEAR image/val
    rm image/val$VALYEAR.zip
fi

if [$TEST -eq 1]
then
    wget http://images.cocodataset.org/zips/test$TESTYEAR.zip -P image/
    unzip image/test$TESTYEAR.zip -d image/ && mv image/test$TESTYEAR image/test
    rm image/test$TESTYEAR.zip
fi
