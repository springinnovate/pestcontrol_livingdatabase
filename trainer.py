import sys
import numpy
import random


from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # replace with your secret key

WEIGHTS = None

def process_lines():
    print('making a new set of lines')
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
    return weights, valid_lines
        # while True:
        #     line = random.choices(valid_lines, weights)[0]
        #     elements = line.split(',')
        #     print(line)
        #     print(f'{elements[1]}\n{elements[2]}\nRank out of 3 best, 2 ok, 1 bad, 0 nonsense')
        #     var = getch()
        #     if var == 'q':
        #         break
        #     train_file.write(f'{var},{line[1:]}')
        #     train_file.flush()

@app.route('/')
def index():
    if 'valid_lines' not in session:
        session['weights'], session['valid_lines'] = process_lines()
    line = random.choices(session['valid_lines'], session['weights'])[0]
    index = session['valid_lines'].index(line)
    session['valid_lines'].pop(index)
    session['weights'].pop(index)
    elements = line.split(',')
    user_line = f'{elements[1]}<br/>{elements[2]}'
    print(user_line)
    return render_template(
        'index.html',
        line=line,
        val_1=elements[1].replace('"', ''),
        val_2=elements[2].replace('"', ''),
        elements_left=len(session['valid_lines'])
        )

@app.route('/rate', methods=['POST'])
def rate():
    rating = request.form.get('rating')
    line = request.form.get('line')
    with open('train_file.txt', 'a') as train_file:
        train_file.write(f'{rating},{line[1:]}\n')
    return redirect(url_for('index'))


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
    #name = input('Enter your name/id: ')
    train_file = open(f'modified_training.csv', 'a')
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
            print(f'\n{elements[1]}\n{elements[2]}\nRank out of 3 best, 2 ok, 1 bad, 0 nonsense')
            var = getch()
            if var == 'q':
                break
            train_file.write(f'{var},{line[1:]}',)
            train_file.flush()

if __name__ == '__main__':
    main()
