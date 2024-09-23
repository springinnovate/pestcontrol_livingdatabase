import sys
import numpy
import random


def getch():
    try:
        # Windows
        if sys.platform.startswith('win'):
            import msvcrt
            return msvcrt.getch().decode('utf-8')
        # Unix-based
        else:
            import termios
            import tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
    except Exception as e:
        print(e)


def main():
    name = input('Hello, what is your name: ')
    train_file = open(f'{name}_modified_training.csv', 'a')
    print(f'ok {name} here we go!')
    with open('training.csv', 'r') as file:
        lines = file.readlines()
        weights = []
        valid_lines = []
        for line in lines:
            elements = line.split(',')
            numeric_weights = [float(x) for x in elements[3:]]
            if numpy.average(numeric_weights) < 0.5:
                continue
            metric = sum(numeric_weights)

            weights.append(metric)
            valid_lines.append(line)
        while True:
            line = random.choices(valid_lines, weights)[0]
            elements = line.split(',')
            print(f'\n{elements[1]}\n{elements[2]}\nRank out of: (1) bad, (2) unsure, (3) match')
            var = getch()
            if var == 'q':
                break
            train_file.write(f'{var},{line[1:]}',)
            train_file.flush()


if __name__ == '__main__':
    main()
