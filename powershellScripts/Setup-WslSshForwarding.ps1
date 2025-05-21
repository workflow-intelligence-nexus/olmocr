<#
.SYNOPSIS
    Configures SSH port forwarding from Windows to WSL2
.DESCRIPTION
    This script sets up port forwarding and firewall rules to allow SSH access to WSL2
    from external machines. Must be run as Administrator.
.NOTES
    File Name      : Setup-WslSshForwarding.ps1
    Author         : System Administrator
    Prerequisite   : PowerShell 5.1 or later, Administrator privileges
#>

# Requires admin rights
#Requires -RunAsAdministrator

param (
    [int]$WindowsPort = 2222,
    [int]$WslPort = 22,
    [string]$RuleName = "WSL2 SSH"
)

function Write-Status {
    param([string]$Message, [string]$Status)
    $time = Get-Date -Format "HH:mm:ss"
    $statusText = if ($Status) { "[$Status] " } else { "" }
    Write-Host "[$time] $statusText$Message"
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check if running as admin
if (-not (Test-IsAdmin)) {
    Write-Error "This script must be run as Administrator"
    exit 1
}

Write-Status "Setting up WSL2 SSH port forwarding..."

try {
    # Get WSL IP address
    $wslIp = bash -c "hostname -I | awk '{print `$1}'"
    if (-not $wslIp) {
        throw "Could not determine WSL2 IP address. Is WSL2 running?"
    }
    
    Write-Status "WSL2 IP address: $wslIp" "INFO"
    Write-Status "Forwarding port $WindowsPort on Windows to port $WslPort on WSL2 ($wslIp)" "INFO"

    # Remove existing port proxy if it exists
    $existingProxy = netsh interface portproxy show v4tov4 | Select-String ":$WindowsPort\s+"
    if ($existingProxy) {
        Write-Status "Removing existing port proxy on port $WindowsPort" "INFO"
        netsh interface portproxy delete v4tov4 listenport=$WindowsPort | Out-Null
    }

    # Add port forwarding
    netsh interface portproxy add v4tov4 listenport=$WindowsPort listenaddress=0.0.0.0 connectport=$WslPort connectaddress=$wslIp
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to set up port forwarding"
    }

    # Configure Windows Firewall
    $firewallRule = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    if ($firewallRule) {
        Write-Status "Updating existing firewall rule: $RuleName" "INFO"
        Remove-NetFireWallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    }

    Write-Status "Creating firewall rule: $RuleName" "INFO"
    New-NetFirewallRule -DisplayName $RuleName `
        -Direction Inbound `
        -LocalPort $WindowsPort `
        -Protocol TCP `
        -Action Allow `
        -Profile Any | Out-Null

    # Verify the setup
    $forwarding = netsh interface portproxy show v4tov4 | Select-String ":$WindowsPort\s+"
    $firewallEnabled = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | 
                      Where-Object { $_.Enabled -eq 'True' }

    if ($forwarding -and $firewallEnabled) {
        $localIp = (Get-NetIPAddress -AddressFamily IPv4 | 
                   Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -ne $wslIp } | 
                   Select-Object -First 1).IPAddress
        
        Write-Status "`n=== Setup Complete ===" "SUCCESS" -ForegroundColor Green
        Write-Status "SSH Access Information:" "INFO" -ForegroundColor Cyan
        Write-Status "  Host: $localIp" "INFO"
        Write-Status "  Port: $WindowsPort" "INFO"
        Write-Status "  Username: YourWslUsername" "INFO"
        Write-Status "  Command: ssh -p $WindowsPort YourWslUsername@$localIp" "INFO"
        Write-Status "`nNote: Replace 'YourWslUsername' with your actual WSL username" "NOTE" -ForegroundColor Yellow
    } else {
        Write-Status "Setup completed but verification failed. Please check the configuration." "WARNING" -ForegroundColor Yellow
    }

} catch {
    Write-Status "Error: $_" "ERROR" -ForegroundColor Red
    exit 1
}

# Display current port forwarding rules
Write-Status "`nCurrent port forwarding rules:" "INFO" -ForegroundColor Cyan
netsh interface portproxy show v4tov4

# Display firewall rule status
Write-Status "`nFirewall rule status:" "INFO" -ForegroundColor Cyan
Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | 
    Select-Object DisplayName, Enabled, Direction, Action | Format-Table -AutoSize