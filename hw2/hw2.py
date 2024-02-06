import math
import sys


def get_parameter_vectors():
    """
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    """
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0] * 26
    s = [0] * 26

    with open("e.txt", encoding="utf-8") as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char) - ord("A")] = float(prob)
    f.close()

    with open("s.txt", encoding="utf-8") as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord("A")] = float(prob)
    f.close()

    return (e, s)


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    X = dict()
    X = {chr(idx): 0 for idx in range(ord("A"), ord("Z") + 1)}
    with open(filename, encoding="utf-8") as f:
        data = f.read()

    for char in data:
        if char.isalpha():
            char = char.upper()
            if char in X:
                X[char] += 1

    return X


def output():
    """
    This function generates the output for the 4 questions

    Returns: Null
    """
    p_e = 0.6
    p_s = 0.4

    X = shred("letter.txt")
    e, s = get_parameter_vectors()

    f_e = math.log(p_e)
    f_s = math.log(p_s)

    for x_i, e_i, s_i in zip(X.values(), e, s):
        f_e += x_i * math.log(e_i)
        f_s += x_i * math.log(s_i)

    diff = f_s - f_e
    if diff >= 100:
        english_given_X = 0
    elif diff <= -100:
        english_given_X = 1
    else:
        english_given_X = 1 / (1 + math.exp(diff))
        english_given_X = round(english_given_X, 4)

    print("Q1")
    for k, v in X.items():
        print("{} {}".format(k, v))

    print("Q2")
    print("{:.4f}".format(X["A"] * math.log(e[0])))
    print("{:.4f}".format(X["A"] * math.log(s[0])))

    print("Q3")
    print("{:.4f}".format(f_e))
    print("{:.4f}".format(f_s))

    print("Q4")
    print("{:.4f}".format(english_given_X))


if __name__ == "__main__":
    output()
