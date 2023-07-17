
   % Load image files
    image_folder = 'D:\Karşıdan İnd\BOUN Dersler\AAA S8\EE 475\Term Project\Datasets\MD1539'; % Specify the path to your image folder
    image_files = dir(fullfile(image_folder, '*.jpg'));
    num_images = length(image_files);

    % Initialize data structures
    labels = zeros(num_images, 1);

    % Extract labels and features for a sample image to determine the size of the HOG feature vector
    img_sample = imread(fullfile(image_folder, image_files(1).name));
    hog_features_sample = compute_hog_features(img_sample);
    num_features = length(hog_features_sample);

    % Preallocate feature_matrix
    feature_matrix = zeros(num_images, num_features);

    % Extract labels and features
    for i = 1:num_images
        [age, ~] = extract_labels(image_files(i).name);
        labels(i) = age;

        img = imread(fullfile(image_folder, image_files(i).name));
        hog_features = compute_hog_features(img);
        feature_matrix(i, :) = hog_features;
    end

   % Apply PCA for feature reduction
    % Set the desired number of features after reduction
  num_reduced_features = 100;  % This value should be determined based on your own data analysis
    [coeff, score, ~, ~, explained] = pca(feature_matrix);
   feature_matrix = score(:, 1:num_reduced_features);
    
    
    % Split data into training and testing sets
    partition = cvpartition(labels, 'Holdout', 0.2);
    training_data = feature_matrix(partition.training,:);
    training_labels = labels(partition.training);
    test_data = feature_matrix(partition.test,:);
    test_labels = labels(partition.test);

    % Train the random forest regressor
    num_trees = 50;
    rf_model = TreeBagger(num_trees, training_data, training_labels, 'Method', 'regression');

    % Test the regressor
    predictions = predict(rf_model, test_data);
    mean_absolute_error = mean(abs(predictions - test_labels));
    fprintf('Mean Absolute Error: %.2f\n', mean_absolute_error);
