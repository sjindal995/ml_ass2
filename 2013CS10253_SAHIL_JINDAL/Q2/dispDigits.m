function dispDigits(data_file,digit,example_index)
	t = load(data_file,digit);
    t = t.(digit);
    image(reshape(t(example_index,:),[28,28])');
end