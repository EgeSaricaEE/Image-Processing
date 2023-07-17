function hog_features = compute_hog_features(image)
    cell_size = [8, 8];
    block_size = [2, 2];
    num_bins = 9;
     if size(image, 3) == 3
        % Convert to grayscale
        img_gray = rgb2gray(image);
    else
        % Image is already in grayscale
        img_gray = image;
    end
    
    hog_features = extractHOGFeatures(img_gray, 'CellSize', cell_size, ...
                                      'BlockSize', block_size, ...
                                      'NumBins', num_bins);
end