%% Parameters
n_images = 5; % amount of examples of each class
im_size = 100; % (px) size of the examples
color_right = [0 255 0];
color_wrong = [255 0 0];
thickness = 2;
images_path = '/Users/Victor/Documents/ETSETB/4B/TFG/Twitter_dataset/';

%% Constants
NEGATIVE = 0;
POSITIVE = 1;

%% Read data from txt files
[path_p,label_p,score_p]=textread('positive_GT.txt','%s %d %f');
[path_n,label_n,score_n]=textread('negative_GT.txt','%s %d %f');

%% Create base image
S_height = n_images*(im_size+3*thickness) + 2*thickness; % height of base image
S_width = 2*(im_size+2*thickness) + thickness; % width of base image (2 columns)
G = 255*ones(S_height, S_width, 3, 'uint8');

%% Sort scores
[dummy,index_p]=sort(score_p,'descend');
[dummy,index_n]=sort(score_n,'descend');

%% Create composition
for column=0:1
    for i=1:n_images
        if column==POSITIVE
            label = label_p(i);
            score = score_p(i);
            path = cast(path_p(i),'char');
        else
            label = label_n(i);
            score = score_n(i);
            path = cast(path_n(i),'char');
        end
        % Load and resize image
        img = imread(strcat(images_path,path));
        img = imresize(img, [im_size, im_size]);
        % Location of i-th image
        x = column*(im_size+3*thickness) + thickness + 1;
        y = (i-1)*(im_size+3*thickness+1) + thickness + 1;
        % Paint edge
        if column == label
            color = cast(color_right, class(img));
        else
            color = cast(color_wrong, class(img));
        end
        I_color = ones(im_size+2*thickness,im_size+2*thickness,3,'uint8');
        for c=1:3
            I_color(:,:,c) = color(c)*I_color(:,:,c);
        end
        G(y-thickness:y+thickness+im_size-1,x-thickness:x+thickness+im_size-1, :) = I_color;
        % Place i-th image
        G(y:y+im_size-1, x:x+im_size-1, :) = img;           
    end
end
figure('name', 'Top scores')
imshow(G);
title('Negative prediction   |   Positive prediction');