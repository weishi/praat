windowShape$ = "rectangular"
relWidth = 1
preserveTimes = 0
input$ = "P_2_1_1_1_1_"

selectObject: "TextGrid P_2_1_1_1"
numberOfIntervals = Get number of intervals: 1
for j to numberOfIntervals
  selectObject: "TextGrid P_2_1_1_1"
  label$ = Get label of interval: 1, j
  if label$ <> "#"
    num$ = string$: j
    soundpath$ = "/tmp/output/" + input$ + num$ + ".wav"
    gridpath$ = "/tmp/output/" + input$ + num$ + ".GridText"
    startTime = Get start point: 1, j
    endTime = Get end point: 1, j
    appendInfoLine: label$, ", ", startTime, " ~ ", endTime
    Extract part... startTime endTime no
    Write to text file... 'gridpath$'

    appendInfoLine: soundpath$
    selectObject: "Sound P_2_1_1_1"
    Extract part: startTime, endTime, windowShape$, relWidth, preserveTimes 
    Write to WAV file... 'soundpath$'
  endif
endfor
