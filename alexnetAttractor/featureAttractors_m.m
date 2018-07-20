%% add path
addpath('/home/dapello/rnnManifolds/featureData')

%% load data
fc7_full = load('caffenet-relu7_og325.mat');
fc7_full = fc7_full.features;

fc7_occ = load('caffenet-relu7_13k.mat');
fc7_occ = fc7_occ.features;

fc7_full(fc7_full>0) = 1;
fc7_full(fc7_full==0) = -1;

fc7_occ(fc7_occ>0) = 1;
fc7_occ(fc7_occ==0) = -1;

% fc7_full_polarized = fc7.fc7_full_polarized; % polarized fc7 reps of 325 full images
% fc7_occ_polarized = fc7.fc7_occ_polarized;   % polarized fc7 reps of 13k occluded images

%% create hopfield net
disp('train hopfield')
T = double(fc7_full');
net = newhop(T);

%% feed fc7_occ_polarized into hopfield for timesteps and record result in cell 325x1 cell array, each cell has #occluded_imgs x timesteps x features
num_occ = length(fc7_occ);
% num_occ = 10;
timesteps = 256;
occ_trajs = zeros(num_occ,6,4096);

for i = 1:num_occ
    t0 = double(fc7_occ(i,:)');
    y = net({1 timesteps}, {}, {t0});
    y = cell2mat(y)';
    occ_trajs(i,:,:) = y(1:50:end,:);
    disp(i)
end

save('/om/user/dapello/caffenetOccTrajs','occ_trajs','-v7.3')


% individual run, most things converging to object 41?
% i=41; occ_i = fc7_occ_polarized{i}; t0 = double(occ_i(10,:)'); y = sim(net, {1 50}, {}, {t0}); sum(T(:,i)==y{50})
