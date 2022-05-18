model = torch.load(args.model_path, map_location=device)
print(type(model)) # <class 'collections.OrderedDict'>

total_params = 0
total_zeros = 0

for key, value in model.items():
    array = value.numpy()
    zero_cnt = np.count_nonzero(abs(array) < 1e-20)
    print(key, value.size(), zero_cnt/array.size)

    if len(array.shape) > 1:
        print(zero_cnt, array.size)
        total_zeros += zero_cnt
        total_params += array.size

    value[abs(value)<(1e-20)] = 0e0

print('percentage of zeroes', total_zeros / total_params)

torch.save(model, 'new_weights.pth')
