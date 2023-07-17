function [age, gender] = extract_labels(filename)
    split_name = split(filename, '_');
    age = str2double(split_name{1});
    gender = str2double(split_name{2});
end