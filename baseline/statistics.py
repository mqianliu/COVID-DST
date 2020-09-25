import ast

total_cnt = {'cough': 0, 'phlegm': 0, 'breath': 0, 'chest': 0, 'runnynose': 0, 'throat': 0, 'fever': 0,
            'chills': 0, 'pain': 0, 'fatigue': 0, 'headache': 0, 'mood': 0, 'gastric': 0,
            'othersym': 0, 'medication': 0, 'pneumonia': 0,
            'asthma': 0, 'diabetes': 0, 'otherdiagnosis': 0, 'travel': 0, 'exposure': 0, 'age': 0, 'smoking': 0,
            'otherphycon': 0, 'REQUEST': 0, 'action': 0, 'prescription': 0, 'diagnose': 0, 'CT/Xray': 0,
           'otherchecking': 0, 'reqmore': 0, 'answer': 0, 'kb': 0}


if __name__ == '__main__':

    s = open("covid-dst-en.json", 'r', encoding='utf-8').read()

    data = ast.literal_eval(s)
    for sample in data.values():
        for utterance in sample["log"]:
            for slot, value in utterance["slots"].items():
                if len(value) > 0:
                    total_cnt[slot] += 1

    print(total_cnt)
    print(len(total_cnt))
    sorted = {k: v for k, v in sorted(total_cnt.items(), key=lambda item: item[1], reverse = True)}
    print(sorted)





