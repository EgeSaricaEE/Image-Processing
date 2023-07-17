
    % Load image files
    image_folder = 'D:\Karşıdan İnd\BOUN Dersler\AAA S8\EE 475\Term Project\Datasets\SmallData'; % Specify the path to your image folder
    image_files = dir(fullfile(image_folder, '*.jpg'));
    num_images = length(image_files);

    % Initialize data structures
    labels = zeros(num_images, 1);
    feature_matrix = [];

    % Extract labels and features
    for i = 1:num_images
        [~, gender] = extract_labels(image_files(i).name);
        labels(i) = gender;

        img = imread(fullfile(image_folder, image_files(i).name));
        hog_features = compute_hog_features(img);
        feature_matrix = [feature_matrix; hog_features];
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

    % Train the SVM classifier
    svm_model = fitcsvm(training_data, training_labels, 'KernelFunction', 'linear');

    % Test the classifier
    predictions = predict(svm_model, test_data);
    accuracy = sum(predictions == test_labels) / length(test_labels);
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);