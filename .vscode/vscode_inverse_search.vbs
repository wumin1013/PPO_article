Option Explicit

Function QuoteArgument(value)
    QuoteArgument = """" & Replace(value, """", """""") & """"
End Function

If WScript.Arguments.Count < 3 Then
    WScript.Quit 1
End If

Dim shell
Dim codePath
Dim filePath
Dim lineNumber
Dim command

Set shell = CreateObject("WScript.Shell")
codePath = WScript.Arguments(0)
filePath = WScript.Arguments(1)
lineNumber = WScript.Arguments(2)
command = QuoteArgument(codePath) & " -r -g " & QuoteArgument(filePath & ":" & lineNumber)

shell.Run command, 0, False
