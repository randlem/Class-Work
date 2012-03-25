:
: The Ohio Supercomputer Center
: Master .profile script
: 10/25/95
: reviced 4/22/03
:
HOSTNAME=`/bin/uname -n`

PS1="[${HOSTNAME}]\$ "		# Your prompt
EDITOR=pico			# Your preferred editor
VISUAL=${EDITOR}
FCEDIT=${EDITOR}                # K-shell history editor
ENV=${HOME}/.kshrc              # K-shell init file name
DISPLAY=mse61.homelinux.net:1
export HOSTNAME PS1 EDITOR VISUAL FCEDIT DISPLAY ENV

alias nano="pico"

# Test and set UNIX terminal type for interactive sessions
if [ "$ENVIRONMENT" != "BATCH" ]; then
  tset -Q -I
  if [ $? -ne 0 -o "$TERM" = "unknown" ]; then
      echo "Enter UNIX terminal type, \c" 1>&2
      TERM=`tset -I -Q - -m ":?vt100"` export TERM
  fi
fi

#
# Put any host specific options and command below here
#
case "$HOSTNAME" in
origin | mss)				# SGI IRIX
    :
    ;;
coe*)					# Sun Solaris
    :
    ;;
*-login*)				# Linux
    :
    ;;
oscb)					# Cray UNICOS
    :
    ;;
*)					# any other host
    :
    ;;
esac
