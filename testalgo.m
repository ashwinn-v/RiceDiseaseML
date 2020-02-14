function riceDisease()
    Path = 'C:\Users\HP\Desktop\rice leaf\Bacterial leaf blight\';
    [Train, Valn] = getTrainValidationData(Path);
    svmModel = svmtrain(Train(:, 1:2), Train(:, 3));
    predictedLabel = svmclassify(Valn(:, 1:2));
    Acc = 100*sum((predictedLabel == Valn(:, 3)))/length(Valn);
end

function [TrainData, ValidationData] = getTrainValidationData(Path)
    
    Feature = [];Folders = {'normal\', 'disease\'};
    for k = 1:2
        files = dir([Path Folders{k}]); 
        for i = 3:length(files)
            im = imread([Path files(i).name]);
            %imshow(im);
            [h, s, v] = rgb2hsv(im);
            imb = im2bw(im2double(rgb2gray(im)));
            subplot(1, 2, 1); imshow(im); subplot(1, 2, 2); imshow(imb);
            meanC = mean(h(imb));
            stdC  = std(h(imb));
            Feature = [Feature; [meanC stdC k]];
        end
      
    end
    split = 0.7; numTrain = ceil(0.7*(length(files) - 2)); 
    totalP = randperm(length(files)-2);
    TrainData = Feature(totalP(1:numTrain), 1:3);
    ValidationData = Feature(totalP(1+numTrain:end, 1:3));
end
