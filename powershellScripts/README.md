# WSL2 SSH Port Forwarding Setup

This project provides a PowerShell script to configure SSH access to WSL2 (Windows Subsystem for Linux 2) from external machines by setting up port forwarding and Windows Firewall rules.

## 📋 Prerequisites

- Windows 10/11 with WSL2 installed
- PowerShell 5.1 or later
- Administrator privileges
- SSH server installed and running in WSL2

## 🚀 Quick Start

1. **Save the script** as `Setup-WslSshForwarding.ps1`
2. **Run as Administrator**:
   ```powershell
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\Setup-WslSshForwarding.ps1


## 🔧 Configuration

### Script Parameters

| Parameter      | Default      | Description                                 |
|---------------|--------------|---------------------------------------------|
| -WindowsPort  | 2222         | Port on Windows to listen on                |
| -WslPort      | 22           | Port in WSL2 that SSH server is running on  |
| -RuleName     | "WSL2 SSH"   | Name for the Windows Firewall rule          |

#### Example with custom ports

```powershell
.\Setup-WslSshForwarding.ps1 -WindowsPort 2222 -WslPort 22 -RuleName "WSL2 SSH Access"
```

## 🔍 How It Works
The script performs the following actions:

- Retrieves WSL2 IP address
- Sets up port forwarding from Windows to WSL2
- Configures Windows Firewall to allow incoming connections
- Verifies the setup
- Displays connection information

## 🔄 Usage Examples

**Basic Usage**
```powershell
.\Setup-WslSshForwarding.ps1
```

**Custom Port Configuration**
```powershell
.\Setup-WslSshForwarding.ps1 -WindowsPort 2223 -WslPort 22
```

**Verify Configuration**
```powershell
# Show port forwarding
netsh interface portproxy show v4tov4

# Show firewall rule
Get-NetFirewallRule -DisplayName "WSL2 SSH" | Select-Object DisplayName, Enabled, Direction, Action
```

## 🗑️ Removing the Configuration
To remove the port forwarding and firewall rule:

```powershell
# Remove port forwarding
netsh interface portproxy delete v4tov4 listenport=2222

# Remove firewall rule
Remove-NetFireWallRule -DisplayName "WSL2 SSH"
```
ule -DisplayName "WSL2 SSH"
🔒 Security Considerations
SSH Authentication:
It's recommended to use SSH keys instead of passwords
Consider changing th

## 🔒 Security & Best Practices

- **Firewall:** The script creates a firewall rule allowing access to the specified port. Consider restricting the source IP addresses if not needed from all networks.
- **Port Selection:** Using non-standard ports (not 22) can help reduce automated attacks. Ensure the selected port doesn't conflict with other services.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support
For support, please open an issue or contact the maintainers.

## 🗂️ Directory Structure
```
wsl-ssh-forwarding/
├── .github/
│   └── workflows/
│       └── test.yml
├── docs/
│   └── advanced-usage.md
├── scripts/
│   └── Setup-WslSshForwarding.ps1
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── README.md
└── SECURITY.md
```

