% DEPENDENCIES
%
%   meanshift
%
%

%% CLEAN-UP

clear;
close all;


%% PARAMETERS

% mean shift options
h = 1.0;
%h =  2.12;
optMeanShift.epsilon = 1e-6;
optMeanShift.verbose = true;
optMeanShift.display = true;

%% DATASET 1
dimensions = 2;
number_of_points = 600;
data_file_name = 'data_cuda.bin';
labels_file_name = 'labels_cuda.bin';
results_file_name_shared = 'results_shared.bin';
results_file_name_global = 'results_global.bin';

%% DATASET 2
%dimensions = 7;
%number_of_points = 210;
%data_file_name = 'data_cuda_seeds.bin';
%labels_file_name = 'labels_cuda_seeds.bin';
%results_file_name_shared = 'results_shared_seeds.bin';
%results_file_name_global = 'results_global_seeds.bin';

%% READING DATA
fileID = fopen(labels_file_name);
l = fread(fileID,[1 number_of_points],'integer*4');
fclose(fileID);
l = l.';

fileID = fopen(data_file_name);
x = fread(fileID,[dimensions number_of_points],'double');
fclose(fileID);
x = x.';
x = double(x);

fileID = fopen(results_file_name_shared);
results = fread(fileID,[dimensions number_of_points],'double');
fclose(fileID);
results = results.';

figure('name', 'original_data')
scatter(x(:,1),x(:,2), 8, l);


%% PERFORM MEAN SHIFT

fprintf('...computing mean shift...')

tic;
y = meanshift( x, h, optMeanShift );
tElapsed = toc;

fprintf('DONE in %.2f sec\n', tElapsed);

%% VALIDATING THE RESULTS
ok_points = 0;
wrong_point = 0;

for i = 1:number_of_points
   for j = 1:dimensions
       if abs(y(i,j) - results(i,j)) > optMeanShift.epsilon
           wrong_point = 1;
           break;
       end
   end
   if wrong_point == 0
       ok_points = ok_points + 1;
   end
end

fprintf('Matching points %d out of %d \n', ok_points, number_of_points);

%% SHOW FINAL POSITIONS
if dimensions == 2
    figure('name', 'final_local_maxima_points')
    scatter(y(:,1),y(:,2), 8, l);
    print('cuda_comparisson','-dpng','-r300')
end
