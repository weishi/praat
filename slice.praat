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
numberOfIntervals = Get number of intervals: 1
idx = 1
for j to numberOfIntervals
  selectObject: "TextGrid " + input$
  label$ = Get label of interval: 1, j
  if label$ = ""
    num$ = string$: idx
    soundpath$ = outputDir$ + num$ + ".wav"
    gridpath$ =  outputDir$ + num$ + ".GridText"
    startTime = Get start point: 1, j
    endTime = Get end point: 1, j
    appendInfoLine: label$, ", ", startTime, " ~ ", endTime
    Extract part... startTime endTime no
    Set interval text: 1, 1, num$
    Write to text file... 'gridpath$'

    appendInfoLine: soundpath$
    selectObject: "Sound " + input$
    Extract part: startTime, endTime, windowShape$, relWidth, preserveTimes 
    Write to WAV file... 'soundpath$'
    idx = idx + 1
  endif
endfor
