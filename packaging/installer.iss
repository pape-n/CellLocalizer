; Inno Setup script for CellLocalizer (formerly Localization Tool) (Windows installer, no admin rights needed).
; Compile after PyInstaller:  iscc /DAppVersion=0.1.0a1 /DAppVersionNum=0.1.0 packaging\installer.iss
; Installs per user to %LOCALAPPDATA%\Programs\CellLocalizer (an update of an older install keeps its folder). Settings in %APPDATA%\LocalizationTool stay on uninstall.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef AppVersionNum
  #define AppVersionNum "0.0.0"
#endif

[Setup]
AppId={{9C4E2B71-3F0A-4D6B-8E15-7A2C5D9B1E44}
AppName=CellLocalizer
AppVersion={#AppVersion}
AppVerName=CellLocalizer {#AppVersion}
VersionInfoVersion={#AppVersionNum}
AppPublisher=Nils Pape
DefaultDirName={autopf}\CellLocalizer
DefaultGroupName=CellLocalizer
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=CellLocalizer-{#AppVersion}-Windows-Setup
UninstallDisplayIcon={app}\CellLocalizer.exe
SetupIconFile=icon.ico
UninstallDisplayName=CellLocalizer
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\CellLocalizer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; remove files of an older version before copying the new one
Type: filesandordirs; Name: "{app}\_internal"
; versions before 2.0.1 were called Localization Tool: their shortcuts and program file
Type: files; Name: "{autoprograms}\Localization Tool.lnk"
Type: files; Name: "{autoprograms}\Localization Tool (safe graphics).lnk"
Type: files; Name: "{autodesktop}\Localization Tool.lnk"
Type: files; Name: "{app}\LocalizationTool.exe"

[Icons]
Name: "{autoprograms}\CellLocalizer"; Filename: "{app}\CellLocalizer.exe"
Name: "{autoprograms}\CellLocalizer (safe graphics)"; Filename: "{app}\CellLocalizer.exe"; Parameters: "--software-opengl"; Comment: "For Remote Desktop or when the image stays black"
Name: "{autodesktop}\CellLocalizer"; Filename: "{app}\CellLocalizer.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\CellLocalizer.exe"; Description: "{cm:LaunchProgram,CellLocalizer}"; Flags: nowait postinstall skipifsilent
