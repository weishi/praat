windowShape$ = "rectangular"
relWidth = 1
preserveTimes = 0
input$ = "65_s2_7"
inputDir$ = ""

Read from file: inputDir$ + input$+ ".TextGrid"
Read from file: inputDir$ + input$+ ".wav"
selectObject: "TextGrid " + input$
Insert interval tier... 2 'merged'
numberOfIntervals = Get number of intervals: 1
for j to numberOfIntervals
  selectObject: "TextGrid " + input$
  label$ = Get label of interval: 1, j
  if label$ = "s"
    num$ = string$: j
    startTime = Get start point: 1, j
    endTime = Get end point: 1, j
    appendInfoLine: label$, ", ", startTime, " ~ ", endTime
    midTime = (startTime + endTime) / 2
    Insert boundary... 2 midTime
    appendInfoLine: label$, ", ", midTime
  endif
endfor
numberOfIntervals = Get number of intervals: 2
for j to numberOfIntervals
  selectObject: "TextGrid " + input$
  num$ = string$: j-1
  if j > 1 && j < numberOfIntervals
    Set interval text: 2, j, num$
  endif
endfor
selectObject: "TextGrid " + input$
gridPath$ = inputDir$ + input$ + "_new.TextGrid"
appendInfoLine: gridPath$
Write to text file... 'gridPath$'
