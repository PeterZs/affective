%% Parameters
images_folder = '/imatge/vcampos/work/twitter_dataset/images/';
features_folder = '/imatge/vcampos/work/twitter_dataset/feature_maps_mat/test1/conv5/';
ground_truth_file = '/imatge/vcampos/work/twitter_dataset/feature_maps_mat/test1.txt';
unitID = 49;                    % conv5 unit
N_images = 5;
first_image = 2;

%% Load text file
images = textread(ground_truth_file, '%s');


%% generate uniform receptive field
RFsize = 65;                    % the average actual size of conv5, you could change it to get a tighter segmentation
para.gridScale = [13 13];       % conv5 of alexNet feature map
para.imageScale = [227 227];    % the input image size
para.RFsize = [RFsize RFsize];  
para.plotPointer = 0;           % whether to show the generated RF
maskRF = generateRF( para);



thresholdSegmentation = 0.5;    % segmentationthreshold

fig = figure('visible','on');

for j=1:+N_images
    i = first_image+j;
    curImg = imread([images_folder cast(images(i),'char') '.jpg']);
    curImg = im2double(imresize(curImg,para.imageScale));
    curFeatureMap = load([features_folder cast(images(i),'char') '.mat']); % the extracted feature map for unitID at conv5 of places-CNN.
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
saveas(fig,'/imatge/vcampos/work/fig','jpg');

