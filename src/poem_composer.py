'''
Compose new poem using Markov model.
'''
import numpy as np


def read_data():
    poem = open("../data/nguyen-binh.txt").read()
    poem += open("../data/truyen_kieu.txt").read()

    lines = poem.lower().split("\n")

    return lines


def create_frequency_matrix(lines):
    state_transition_matrix = dict()
    second = dict()
    initial = dict()

    for line in lines:
        line += ' end'

        tokens = line.split(' ')

        for i in range(len(tokens)):

            if i == 0:
                # the first word
                s_0 = tokens[i]
                initial[s_0] = initial.get(s_0, 0) + 1

            elif i == 1:
                # the second word
                s_0 = tokens[i]
                s_1 = tokens[i - 1]

                if s_1 not in second:
                    second[s_1] = dict()

                second[s_1][s_0] = second[s_1].get(s_0, 0) + 1

            else:
                s_0 = tokens[i]
                s_1 = tokens[i - 1]
                s_2 = tokens[i - 2]

                key = (s_2, s_1)

                if key not in state_transition_matrix:
                    state_transition_matrix[key] = dict()

                state_transition_matrix[key][s_0] = state_transition_matrix[key].get(s_0, 0) + 1

    return initial, second, state_transition_matrix


def convert2prob(dict):
    """
    Input: dâng {'lên': 2, 'tranh': 2}

    Output: dâng {'lên': 0.5, 'tranh': 0.5}
    :param dict: a dictionary
    :return:
    """
    sum = 0

    for k, v in dict.items():
        sum += v

    for k, v in dict.items():
        dict[k] = v / sum

    return dict


def sample(dict):
    """
    Input: dâng {'lên': 2, 'tranh': 2}

    Output: 'lên' or 'tranh'
    :param dict: a dictionary
    :return: a sample from dictionary
    """
    pvals = []
    word_options = []

    for k, v in dict.items():
        pvals.append(v)
        word_options.append(k)

    sample = np.random.multinomial(1, pvals, size=1)
    word = word_options[np.argmax(sample)]
    return word


if __name__ == '__main__':
    lines = read_data()
    initial, second, state_transition_matrix = create_frequency_matrix(lines)

    # convert frequency matrix to probability matrix
    convert2prob(initial)

    for _, v in second.items():
        convert2prob(v)

    for _, v in state_transition_matrix.items():
        convert2prob(v)

    # print for testing
    '''
    for k, v in bigram.items():
        print(k, v)

    for k, v in second.items():
        print(k, v)

    print(initial)
    '''
    # compose a poem
    num_of_lines = 10
    new_poem = []

    for i in range(num_of_lines):
        words = []

        s_2 = sample(initial)
        words.append(s_2)

        s_1 = sample(second[s_2])
        words.append(s_1)

        # generate the remaining words
        while (True):
            t = sample(state_transition_matrix[(s_2, s_1)])

            if t == 'end':
                break;
            else:
                words.append(t)
                s_2 = s_1
                s_1 = t

        # save line of new poem
        new_poem.append(' ' .join(words))

    print('\n'.join(new_poem))
