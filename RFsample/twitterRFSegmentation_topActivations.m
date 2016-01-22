%% Parameters
images_folder = '/imatge/vcampos/work/twitter_dataset/images/';
features_folder = '/imatge/vcampos/work/twitter_dataset/feature_maps_mat/test1/conv5/';
ground_truth_file = '/imatge/vcampos/work/twitter_dataset/feature_maps_mat/test1.txt';
unitSelect_file = 'unitID.txt';
N_images = 5;


%% Load unitID from text file
unitID = textread(unitSelect_file, '%d');   % conv5 unit
disp(['UnitID: ' num2str(unitID)])


%% Load text file
images = textread(ground_truth_file, '%s');
maxValFeatureMaps = [];


%% Load feature maps for unitID and compute their sum
disp('Loading feature maps...')
for i=1:size(images,1)
    curFeatureMap = load([features_folder cast(images(i),'char') '.mat']); % the extracted feature map for unitID at conv5
    curFeatureMap = curFeatureMap.featureMap(unitID,:,:);  % select activation from unitID
    maxValFeatureMaps = [maxValFeatureMaps max(curFeatureMap(:))];
end


%% Sort images so only the ones with the top activations are displayed
disp('Sorting top activations')
[values, sorted_indexes] = sort(maxValFeatureMaps, 'descend');


%% generate uniform receptive field
RFsize = 65;                    % the average actual size of conv5, you could change it to get a tighter segmentation
para.gridScale = [13 13];       % conv5 of alexNet feature map
para.imageScale = [227 227];    % the input image size
para.RFsize = [RFsize RFsize];  
para.plotPointer = 0;           % whether to show the generated RF
maskRF = generateRF( para);



thresholdSegmentation = 0.5;    % segmentationthreshold

%% Estimate RF for the images that generated the highest unitID activations
disp('Estimating RF...')
for j=1:+N_images
    curImg = imread([images_folder cast(images(sorted_indexes(j)),'char') '.jpg']);
    curImg = im2double(imresize(curImg,para.imageScale));
    curFeatureMap = load([features_folder cast(images(sorted_indexes(j)),'char') '.mat']); % the extracted feature map for unitID at conv5
    curFeatureMap = curFeatureMap.featureMap(unitID,:,:);
    curFeature_vectorized = curFeatureMap(:);
    maxValue = max(curFeature_vectorized);
    IDX_max = find(curFeature_vectorized>maxValue * thresholdSegmentation);
    curMask = squeeze(sum(maskRF(IDX_max,:,:),1));
    curMask(curMask>0) = 1;

    IDX_region = find(curMask>0);
    curSegmentation = repmat(curMask,[1 1 3]).*curImg+0.2*(1- repmat(curMask,[1 1 3])).*curImg;

    imwrite(curSegmentation, ['/imatge/vcampos/work/RF_results/ImageNet/unit' num2str(unitID) '_' num2str(j) '.jpg']);
end

disp('Done!')
