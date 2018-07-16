tic
%% create 13k partially occluded images from original 325.
filepath = 'data/data_occlusion_klab325v2.mat';
data = load(filepath);

% sort by object id
[data, sort_indx] = sortrows(data.data, {'pres'});

% get info for creating occluded images
% black = data.black;
pres = data.pres;

% and object class (fruit, face, etc)
class = data.truth;

% walk through each identity/occlusion value, writing the occluded images
% organize the images as:
% data is saved in a cell format with dimension matching the number of "objects"?
% ie data{i}, where i goes from 1 to P where P is the number of objects (325). 
% data{i} should be in the dimension of M by N, M the number of images per object, N is the feature dimension 
% also inclue called "category(i)" (truth) and "category_name{i}", 
% where category(i) specifies which category each object class corresponds to in terms of integer number 
% and category_name should have the name of the category. 
% There should also be a variable called "object_name{i}" which corresponds to the object class name. 

% img_count = length(pres);
obj_count = 325;
im_size = 227;

% random sample of image features (pixels) same for all images
index_rand = randperm(154587,5000);

dataset = cell(obj_count,4);
base_count = 0;
for i = 1:obj_count
    % number of occluded images for current object (varies between 42 and 16?)
    n_inst = sum(pres == i);
    
    % array to hold series of occluded images in
    imgs = zeros(n_inst, im_size, im_size, 3);
    
    % iterate through all instances for current object
    for j = 1:n_inst
        % index goes from 1:13000 as base_count iterates
        index = base_count+j;
        
        % create, prepare, and store actual image
        img = createOccludedImage(sort_indx(index),false);
        img = prepareGrayscaleImage(img);
        imgs(j,:,:,:) = img;
    end
    
    flat_imgs = reshape(imgs,n_inst,[],1);
    flat_imgs_sampled = flat_imgs(:,index_rand);
    dataset{i,1} = imgs;
    dataset{i,2} = flat_imgs_sampled;
    dataset{i,3} = class(index);
    dataset{i,4} = pres(index);
    
    % iterate basecount
    base_count = base_count + n_inst;
    disp(base_count)
end

save('occludedImgs13000.mat','dataset','-v7.3')

%
toc