# /Applications/Praat.app/Contents/MacOS/Praat --run "~/Downloads/wuren/wav/slicing.praat"
form Test command line calls
    sentence DirBase ""
    sentence Input ""
endform

windowShape$ = "rectangular"
relWidth = 1
preserveTimes = 0
# input$ = "01s1_7"
inputDir$ = dirBase$ + "/"
outputDir$ = dirBase$ + "/" + input$ + "/"

createDirectory: outputDir$

Read from file: inputDir$ + input$+ ".TextGrid"
Read from file: inputDir$ + input$+ ".wav"
selectObject: "TextGrid "+ input$
Insert interval tier... 2 'merged'
numberOfIntervals = Get number of intervals: 1
for j to numberOfIntervals
  selectObject: "TextGrid " + input$
  label$ = Get label of interval: 1, j
  if label$ <> ""
    startTime = Get start point: 1, j
    endTime = Get end point: 1, j
    midTime = (startTime + endTime) / 2
    Insert boundary... 2 midTime
    # appendInfoLine: label$, ", ", startTime, " ~ ", midTime, " ~ ", endTime
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
mergedGridPath$ = inputDir$ + input$ + "_new.TextGrid"
appendInfoLine: mergedGridPath$
Write to text file... 'mergedGridPath$'

numberOfIntervals = Get number of intervals: 2
for j to numberOfIntervals
  selectObject: "TextGrid " + input$
  num$ = Get label of interval: 2, j
  if num$ <> ""
    startTime = Get start point: 2, j
    endTime = Get end point: 2, j
    soundpath$ = outputDir$ + num$ + ".wav"
    gridpath$ =  outputDir$ + num$ + ".GridText"
    Extract part... startTime endTime no
    Write to text file... 'gridpath$'

    # appendInfoLine: soundpath$
    selectObject: "Sound " + input$
    Extract part: startTime, endTime, windowShape$, relWidth, preserveTimes 
    Write to WAV file... 'soundpath$'
  endif
endfor
