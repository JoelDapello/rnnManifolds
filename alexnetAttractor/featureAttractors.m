%% add path
addpath('/home/dapello/rnnManifolds/featureData')

%% load data
fc7 = load('vgg15FullAndOccPol_325.mat');
 
fc7_full_polarized = fc7.fc7_full_polarized; % polarized fc7 reps of 325 full images
fc7_occ_polarized = fc7.fc7_occ_polarized;   % polarized fc7 reps of 13k occluded images

%% create hopfield net
disp('train hopfield')
T = double(fc7_full_polarized');
net = newhop(T);

%% feed fc7_occ_polarized into hopfield for timesteps and record result in cell 325x1 cell array, each cell has #occluded_imgs x timesteps x features
% num_objs = length(fc7_occ_polarized);
num_objs = 325;
timesteps = 256;
occ_trajs = cell(num_objs,1);
for i = 1:num_objs
    occ_i = fc7_occ_polarized{i};
    occ_size = size(occ_i);
    num_occ = occ_size(1);
    features = occ_size(2);

    occ_i_trajs = zeros(num_occ,6,features);
    for j = 1:12:num_occ
        t0 = double(occ_i(j,:)');
        y = net({1 timesteps}, {}, {t0});
	y = cell2mat(y)';
        occ_i_trajs(j,:,:) = y(1:50:end,:);
    end
    occ_trajs{i} = occ_i_trajs;
    disp(i)
end

save('/om/user/dapello/vgg16OccTrajs','occ_trajs','-v7.3')


% individual run, most things converging to object 41?
% i=41; occ_i = fc7_occ_polarized{i}; t0 = double(occ_i(10,:)'); y = sim(net, {1 50}, {}, {t0}); sum(T(:,i)==y{50})
