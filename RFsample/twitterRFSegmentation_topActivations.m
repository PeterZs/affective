%% Parameters
images_folder = '/imatge/vcampos/work/twitter_dataset/images/';
features_folder = '/imatge/vcampos/work/twitter_dataset/feature_maps_mat/test1/conv5/';
ground_truth_file = '/imatge/vcampos/work/twitter_dataset/feature_maps_mat/test1.txt';
unitSelect_file = 'unitID.txt';
N_images = 5;


%% Load unitID from text file
unitID = textread(unitSelect_file, '%d');   % conv5 unit


%% Load text file
images = textread(ground_truth_file, '%s');
maxValFeatureMaps = [];


%% Load feature maps for unitID and compute their sum
for i=1:size(images,1)
    curFeatureMap = load([features_folder cast(images(i),'char') '.mat']); % the extracted feature map for unitID at conv5
    curFeatureMap = curFeatureMap.featureMap(unitID,:,:);  % select activation from unitID
    maxValFeatureMaps = [maxValFeatureMaps max(curFeatureMap(:))];
end


%% Sort images so only the ones with the top activations are displayed
[values, sorted_indexes] = sort(maxValFeatureMaps, 'descending');

fig = figure('visible','off');


%% Estimate RF for the images that generated the highest unitID activations
for j=1:+N_images
    curImg = imread([images_folder cast(images(images(sorted_indexes(j))),'char') '.jpg']);
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

    subplot(1,N_images,j),imshow(curSegmentation);
end

%% Save figure
saveas(fig,'/imatge/vcampos/work/fig','jpg');