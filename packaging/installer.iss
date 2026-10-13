; Inno Setup script for Localization Tool (Windows installer, no admin rights needed).
; Compile after PyInstaller:  iscc /DAppVersion=0.1.0a1 /DAppVersionNum=0.1.0 packaging\installer.iss
; Installs per user to %LOCALAPPDATA%\Programs\Localization Tool. Settings in %APPDATA%\LocalizationTool stay on uninstall.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef AppVersionNum
  #define AppVersionNum "0.0.0"
#endif

[Setup]
AppId={{9C4E2B71-3F0A-4D6B-8E15-7A2C5D9B1E44}
AppName=Localization Tool
AppVersion={#AppVersion}
AppVerName=Localization Tool {#AppVersion}
VersionInfoVersion={#AppVersionNum}
AppPublisher=Nils Pape
DefaultDirName={autopf}\Localization Tool
DefaultGroupName=Localization Tool
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=LocalizationTool-{#AppVersion}-Windows-Setup
UninstallDisplayIcon={app}\LocalizationTool.exe
SetupIconFile=icon.ico
UninstallDisplayName=Localization Tool
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
Source: "..\dist\LocalizationTool\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; remove files of an older version before copying the new one
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\Localization Tool"; Filename: "{app}\LocalizationTool.exe"
Name: "{autoprograms}\Localization Tool (safe graphics)"; Filename: "{app}\LocalizationTool.exe"; Parameters: "--software-opengl"; Comment: "For Remote Desktop or when the image stays black"
Name: "{autodesktop}\Localization Tool"; Filename: "{app}\LocalizationTool.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\LocalizationTool.exe"; Description: "{cm:LaunchProgram,Localization Tool}"; Flags: nowait postinstall skipifsilent
