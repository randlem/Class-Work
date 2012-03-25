#
# The Ohio Supercomputer Center
# Master .login script for MPP systems
# 10/25/95
#
setenv HOSTNAME `/bin/uname -n`

set prompt="[${HOSTNAME}]\% "		# Your prompt

setenv EDITOR vi			# Your editor
setenv VISUAL ${EDITOR}

# Test and set UNIX terminal type
tset -Q -I
if ( $status != 0 || "$TERM" == "unknown" ) then
    echo -n "Enter UNIX terminal type, "
    eval `tset -I -Q -s -m ":?vt100" | fgrep ' TERM '`
endif

if      ($HOSTNAME =~ oscx*) then	# Convex Exemplar
    :
else if ($HOSTNAME =~ oscsp*) then	# IBM SP2
    :
else if ($HOSTNAME =~ sgipc*) then	# SGI Power Challenge
    :
else if ($HOSTNAME =~ origin) then	# SGI Origin 2000
    :
else if ($HOSTNAME =~ OSCA) then	# Cray UNICOS
    :
else if ($HOSTNAME =~ oscj) then	# Cray UNICOS
    :
else if ($HOSTNAME =~ T3E) then		# Cray UNICOS
    :
else					# any other host
    :
endif
