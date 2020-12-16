import tkinter as tk
from math import exp

root = tk.Tk()

canvas1 = tk.Canvas(root, width=400, height=620, relief='raised')
canvas1.pack()

label1 = tk.Label(root, text='Predict the winner')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)

label2 = tk.Label(root, text='Please type the team name and year for the first team')
label2.config(font=('helvetica', 10))
canvas1.create_window(200, 100, window=label2)

team1 = tk.Entry(root)
canvas1.create_window(100, 140, window=team1)

year1 = tk.Entry(root)
canvas1.create_window(300, 140, window=year1)

label2 = tk.Label(root, text='Please type the team name and year for the second team')
label2.config(font=('helvetica', 10))
canvas1.create_window(200, 180, window=label2)

team2 = tk.Entry(root)
canvas1.create_window(100, 220, window=team2)

year2 = tk.Entry(root)
canvas1.create_window(300, 220, window=year2)


def getSquareRoot():
    team1name = team1.get()
    team_arr = (team_season_old.team == str(team1name)).values

    year1name = year1.get()
    year_arr = (team_season_old.year == int(year1name)).values

    count = 0
    index = -1
    for value in team_arr:
        if (team_arr[count] == True):
            if (year_arr[count] == True):
                index = count
        count = count + 1
    print(index)
    print(team_season_no_wins.values[index].tolist())

    Xnew = np.array([team_season_no_wins.values[index].tolist()])
    LMX = Xnew
    Xnew = scaler_x.transform(Xnew)
    ynew = model.predict(Xnew)
    # invert normalize
    ynew = scaler_y.inverse_transform(ynew)
    Xnew = scaler_x.inverse_transform(Xnew)
    print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

    team2name = team2.get()
    team2_arr = (team_season_old.team == str(team2name)).values

    year2name = year2.get()
    year2_arr = (team_season_old.year == int(year2name)).values

    count = 0
    index2 = -1
    for value in team2_arr:
        if (team2_arr[count] == True):
            if (year2_arr[count] == True):
                index2 = count
        count = count + 1
    print(index2)
    print(team_season_no_wins.values[index2].tolist())

    Xnew2 = np.array([team_season_no_wins.values[index2].tolist()])
    LMX2 = Xnew2
    Xnew2 = scaler_x.transform(Xnew2)
    ynew2 = model.predict(Xnew2)
    # invert normalize
    ynew2 = scaler_y.inverse_transform(ynew2)
    Xnew2 = scaler_x.inverse_transform(Xnew2)
    print("X=%s, Predicted=%s" % (Xnew2[0], ynew2[0]))

    prob1 = exp(float(ynew[0][0])) / (exp(float(ynew[0][0])) + exp(float(ynew2[0][0])))
    print(prob1)

    prob2 = exp(float(ynew2[0][0])) / (exp(float(ynew[0][0])) + exp(float(ynew2[0][0])))
    print(prob2)

    label5 = tk.Label(root, text='For the Neural Network', font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 320, window=label5)

    label3 = tk.Label(root, text='The normalized probability for team one winning is: ' + str(prob1),
                      font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 360, window=label3)

    label4 = tk.Label(root, text='The normalized probability for team two winning is: ' + str(prob2),
                      font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 380, window=label4)

    print("LM prediction is" + str(mullinpredictions[index]))

    LMPred1 = float(mullinpredictions[index])
    LMPred2 = float(mullinpredictions[index2])

    normalizedLM1 = exp(LMPred1) / (exp(LMPred1) + exp(LMPred2))
    print(normalizedLM1)

    normalizedLM2 = exp(LMPred2) / (exp(LMPred1) + exp(LMPred2))
    print(normalizedLM2)

    label8 = tk.Label(root, text='For the Multiple Linear Regression Model', font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 420, window=label8)

    label9 = tk.Label(root, text='The normalized probability for team one winning is: ' + str(normalizedLM1),
                      font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 460, window=label9)

    label10 = tk.Label(root, text='The normalized probability for team two winning is: ' + str(normalizedLM2),
                       font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 480, window=label10)

    finalTeam1 = prob1 + normalizedLM1
    finalTeam2 = prob2 + normalizedLM2

    FinalnormalizedLM1 = exp(finalTeam1) / (exp(finalTeam1) + exp(finalTeam2))
    print(normalizedLM1)

    FinalnormalizedLM2 = exp(finalTeam2) / (exp(finalTeam1) + exp(finalTeam2))
    print(normalizedLM2)

    label11 = tk.Label(root, text='For the Ensemble of Both Models', font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 520, window=label11)

    label12 = tk.Label(root, text='The normalized probability for team one winning is: ' + str(FinalnormalizedLM1),
                       font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 560, window=label12)

    label13 = tk.Label(root, text='The normalized probability for team two winning is: ' + str(FinalnormalizedLM2),
                       font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 580, window=label13)

    winner = ""
    if (float(FinalnormalizedLM1) > float(FinalnormalizedLM2)):
        winner = "The Ensemble result shows that team 1 wins"
    elif (float(FinalnormalizedLM2) > float(FinalnormalizedLM1)):
        winner = "The Ensemble result shows that team 2 wins"

    label16 = tk.Label(root, text=winner, font=('helvetica', 10, 'bold'))
    canvas1.create_window(200, 620, window=label16)


#     logpredictedprob1 = linregr.predict(Xnew)
#     logpredictedprob2 = linregr.predict(Xnew2)

#     label6 = tk.Label(root, text='For the Multiple Linear Regression Model', font=('helvetica', 10, 'bold'))
#     canvas1.create_window(200, 390, window=label5)

#     logprobnorm1 = exp(float(logpredictedprob1)) / (exp(float(logpredictedprob1)) + exp(float(logpredictedprob2)))
#     print(prob1)

#     logprobnorm2 = exp(float(logpredictedprob2)) / (exp(float(logpredictedprob1)) + exp(float(logpredictedprob2)))
#     print(prob2)

#     label3 = tk.Label(root, text='The normalized probability for team one winning is: '+str(logprobnorm1), font=('helvetica', 10, 'bold'))
#     canvas1.create_window(200, 420, window=label3)

#     label4 = tk.Label(root, text='The normalized probability for team two winning is: '+str(logprobnorm2), font=('helvetica', 10, 'bold'))
#     canvas1.create_window(200, 440, window=label4)


#     winner = ''
#     if(prob1>prob2):
#         winner = 'This indicates that the first team has a higher chance of winning'
#     else:
#         winner = 'This indicates that the second team has a higher chance of winning'


#     label5 = tk.Label(root, text=winner, font=('helvetica', 10, 'bold'))
#     canvas1.create_window(200, 380, window=label5)


button1 = tk.Button(text='Run prediction', command=getSquareRoot, bg='brown', fg='white',
                    font=('helvetica', 9, 'bold'))
canvas1.create_window(200, 260, window=button1)

root.mainloop()