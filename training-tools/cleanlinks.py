input_file = 'linksdirty.txt'
output_file = 'links.txt'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        index = line.find('.tar')
        if index != -1:
            tar_name = line[:index + 4]  # Keep '.tar' in the output
            f_out.write(tar_name + '\n')

