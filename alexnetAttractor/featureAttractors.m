function featureAttractors()
load('data/fc7_features')

%% Train
% attractor network
indx = [1,2,5,8,9]; % one fixed point per category

T = double(fc7_train(indx,:)');
net = newhop(T);


%% get hopfield trajectories
features = length(fc7_test(1,:))
samples = length(fc7_test(:,1))
timesteps = 60

% for test set of whole images
fc7_test_hop_trajs = zeros(samples,timesteps,features);
for i = 1:samples
    t0 = double(fc7_test(i,:));
    y = sim(net, {1 timesteps}, {}, {t0'});
    fc7_test_hop_trajs(i,:,:) = cell2mat(y)';
    disp(i) 
    disp(size(t0))
    disp(size(y))
end

% for test set of occluded images
fc7_test_occ_hop_trajs = zeros(samples,timesteps,features);
for i = 1:samples
    t0 = double(fc7_test_occ(i,:));
    y = sim(net, {1 timesteps}, {}, {t0'});
    fc7_test_occ_hop_trajs(i,:,:) = cell2mat(y)';
    disp(i) 
    disp(size(t0))
    disp(size(y))
end

%% save hopfield trajectories
% save('fc7_hop_trajs', fc7_test_hop_trajs, fc7_test_occ_hop_trajs)
save 'fc7_hop_trajs.mat' fc7_test_hop_trajs fc7_test_occ_hop_trajs