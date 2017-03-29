#!/bin/bash
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CONF_FILE=${HOME}/l4t_dfs.conf
RED='\e[0;31m'
GREEN='\e[0;32m'
BLUE='\e[0;34m'
BRED='\e[1;31m'
BGREEN='\e[1;32m'
BBLUE='\e[1;34m'
NC='\e[0m' # No Color

usage()
{
	if [ "$1" != "" ]; then
		echo -e ${RED}"$1"${NC}
	fi

		echo "usage:"

		cat >& 2 <<EOF
		jetson_max_l4t.sh [options]
		options,
		--show                display current settings
		--store [file]        store current settings to a file (default: /home/ubuntu/l4t_dfs.conf)
		--restore [file]      restore saved settings from a file (default: /home/ubuntu/l4t_dfs.conf)
EOF

	exit 0
}

restore()
{
	for conf in `cat $CONF_FILE`; do
		file=`echo $conf | cut -f1 -d :`
		data=`echo $conf | cut -f2 -d :`
		case $file in
			/sys/devices/system/cpu/cpu*/online |\
			/sys/kernel/debug/clock/override*/state )
				if [ `cat $file` -ne $data ]; then
					echo $data > $file
				fi
				;;
			*)
				echo $data > $file
				ret=$?
				if [ ${ret} -ne 0 ]; then
					echo "Error: Failed to restore $file"
				fi
				;;
		esac
	done
}

store()
{
	for file in $@; do
		if [ -e "${file}" ]; then
			echo "$file:`cat $file`" >> $CONF_FILE
		fi
	done
}

do_fan()
{
	# Jetson-TK1 CPU fan is always ON.
	if [ -e /sys/devices/soc0/machine ]; then
		machine=`cat /sys/devices/soc0/machine`
		if [ "${machine}" = "jetson-tk1" ] ; then
			return
		fi
	fi

	if [ ! -w /sys/kernel/debug/tegra_fan/target_pwm ]; then
		echo "Can't access Fan!"
		return
	fi

	case $ACTION in
		show)
			echo "Fan: speed=`cat /sys/kernel/debug/tegra_fan/target_pwm`"
			;;
		store)
			store /sys/kernel/debug/tegra_fan/target_pwm
			;;
		*)
			FAN_SPEED=255
			echo $FAN_SPEED > /sys/kernel/debug/tegra_fan/target_pwm
			;;
	esac
}

do_clusterswitch()
{
	case $ACTION in
		show)
			if [ -d /sys/kernel/cluster ]; then
				ACTIVE_CLUSTER=`cat /sys/kernel/cluster/active`
				echo "CPU Cluster Switching: Active Cluster ${ACTIVE_CLUSTER}"
			else
				echo "CPU Cluster Switching: Disabled"
			fi
			;;
		store)
			if [ -d /sys/kernel/cluster ]; then
				store "/sys/kernel/cluster/immediate"
				store "/sys/kernel/cluster/force"
				store "/sys/kernel/cluster/active"
			fi
			;;
		*)
			if [ -d /sys/kernel/cluster ]; then
				echo 1 > /sys/kernel/cluster/immediate
				echo 0 > /sys/kernel/cluster/force
				echo G > /sys/kernel/cluster/active
			fi
			;;
	esac
}

do_hotplug()
{
	CPU_HOTPLUG_STAT=`cat /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable`

	case $ACTION in
		show)
			echo "CPU HOTPLUG: $CPU_HOTPLUG_STAT"
			echo "Online CPUs: `cat /sys/devices/system/cpu/online`"
			for folder in /sys/devices/system/cpu/cpu[0-9]; do
				if [ -e "${folder}/cpufreq/scaling_cur_freq" ]; then
					CPU=`echo ${folder} | cut -c 25-`
					echo "$CPU: `cat ${folder}/cpufreq/scaling_cur_freq`"
				fi
			done
			;;
		store)
			store "/sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable"
			for file in /sys/devices/system/cpu/cpu*/online; do
				store $file
			done
			;;
		*)
			echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
			for file in /sys/devices/system/cpu/cpu*/online; do
				if [ `cat $file` -eq 0 ]; then
					echo 1 > $file
				fi
			done
	esac
}

do_cpu()
{
	FRQ_GOVERNOR=`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
	CPU_MIN_FREQ=`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq`
	CPU_MAX_FREQ=`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq`
	CPU_CUR_FREQ=`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq`

	case $ACTION in
		show)
			echo "CPU frequency Governor: $FRQ_GOVERNOR"
			echo "CPU MinFreq=$CPU_MIN_FREQ MaxFreq=$CPU_MAX_FREQ CurrentFreq=$CPU_CUR_FREQ"
			;;
		store)
			store "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"

			if [ -d /sys/devices/system/cpu/cpufreq/$FRQ_GOVERNOR ]; then
				store `find /sys/devices/system/cpu/cpufreq/$FRQ_GOVERNOR -type f -perm -g+r`
			fi
			;;
		*)
			echo userspace > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
			echo $CPU_MAX_FREQ >  /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
			;;
	esac
}

do_gpu()
{
	GPU_MIN_FREQ=`cat /sys/kernel/debug/clock/override.gbus/min`
	GPU_MAX_FREQ=`cat /sys/kernel/debug/clock/override.gbus/max`
	GPU_CUR_FREQ=`cat /sys/kernel/debug/clock/override.gbus/rate`
	GPU_FREQ_OVERRIDE=`cat /sys/kernel/debug/clock/override.gbus/state`

	case $ACTION in
		show)
			echo "GPU MinFreq=$GPU_MIN_FREQ MaxFreq=$GPU_MAX_FREQ CurrentFreq=$GPU_CUR_FREQ FreqOverride=$GPU_FREQ_OVERRIDE"
			;;
		store)
			store /sys/kernel/debug/clock/override.gbus/rate
			store /sys/kernel/debug/clock/override.gbus/state
			;;
		*)
			echo $GPU_MAX_FREQ > /sys/kernel/debug/clock/override.gbus/rate
			echo 1 > /sys/kernel/debug/clock/override.gbus/state
			ret=$?
			if [ ${ret} -ne 0 ]; then
				echo "Error: Failed to max GPU frequency!"
			fi
			;;
	esac
}

do_emc()
{
	EMC_MIN_FREQ=`cat /sys/kernel/debug/clock/override.emc/min`
	EMC_MAX_FREQ=`cat /sys/kernel/debug/clock/override.emc/max`
	EMC_CUR_FREQ=`cat /sys/kernel/debug/clock/override.emc/rate`
	EMC_FREQ_OVERRIDE=`cat /sys/kernel/debug/clock/override.emc/state`

	case $ACTION in
		show)
			echo "EMC MinFreq=$EMC_MIN_FREQ MaxFreq=$EMC_MAX_FREQ CurrentFreq=$EMC_CUR_FREQ FreqOverride=$EMC_FREQ_OVERRIDE"
			;;
		store)
			store /sys/kernel/debug/clock/override.emc/rate
			store /sys/kernel/debug/clock/override.emc/state
			;;
		*)
			echo $EMC_MAX_FREQ > /sys/kernel/debug/clock/override.emc/rate
			echo 1 > /sys/kernel/debug/clock/override.emc/state
			;;
	esac
}

check_uptime()
{

if [ -e "/proc/uptime" ]; then
	uptime=`cat /proc/uptime | cut -d '.' -f1`

	if [ $((uptime)) -lt 90 ]; then
		printf "Error: Please run the script after $((90 - uptime)) Seconds, \
\notherwise ubuntu init script may override the clock settings!\n"
		exit -1
	fi
else
	printf "Warning: Could not check system uptime. Please make sure that you run the script 90 Seconds after bootup, \
\notherwise ubuntu init script may override the clock settings!\n"
fi
}

main ()
{
	check_uptime
	while [ -n "$1" ]; do
		case "$1" in
			--show)
				ACTION=show
				;;
			--store)
				[ -n "$2" ] && CONF_FILE=$2
				ACTION=store
				shift 1
				;;
			--restore)
				[ -n "$2" ] && CONF_FILE=$2
				ACTION=restore
				shift 1
				;;
			-h|--help)
				usage
				exit 0
				;;
			*)
				usage "Unknown option: $1"
				exit 1
				;;
		esac
		shift 1
	done

	[ `whoami` != root ] && echo Error: Run this script\($0\) as a root user && exit 1

	case $ACTION in
		store)
			if [ -e "${CONF_FILE}" ]; then
				echo "File $CONF_FILE already exists. Can I overwrite it? Y/N:"
				read answer
				case $answer in
					y|Y)
						rm -f $CONF_FILE
						;;
					*)
						echo "Error: file $CONF_FILE already exists!"
						exit 1
						;;
				esac
			fi
			;;
		restore)
			if [ ! -e "${CONF_FILE}" ]; then
				echo "Error: $CONF_FILE file not found !"
				exit 1
			fi
			restore
			exit 0
			;;
	esac

	do_cpu
	do_hotplug
	do_clusterswitch
	do_gpu
	do_emc
	do_fan
}

main $@
exit 0
