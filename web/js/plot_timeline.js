let searchParams = new URLSearchParams(window.location.search);
param = searchParams.get('log');

kernels_run = "Kernels Run"
gpu2gpu_copies = "GPU2GPU Copies"
cpu2gpu_copies = "CPU2GPU Copies"
evictions = "Evictions"
free_label = "Free"

$.getJSON("api/logs/" + param, function (profiling_data) {

    var compute_time = [];
    var start_times = [];
    var end_times = [];

    groups = []
    for (let i = 0; i < profiling_data.device_logs.length; i++) {

        data = [];
        cur_compute_time = 0;
        cur_start_time = Infinity;
        cur_end_time = 0;

        // add the info about the kernels that were run on this GPU
        kr_dps = [];
        ks = profiling_data.device_logs[i].kernels_stats;
        for (let j = 0; j < ks.length; j++) {
            cur_compute_time += ks[j].end - ks[j].start;
            cur_end_time = Math.max(cur_end_time, ks[j].end);
            dp = {
                timeRange: [ks[j].start * 1e-5, ks[j].end * 1e-5],
                val: j
            }
            kr_dps.push(dp);
        }
        kr_dps = {
            label: kernels_run,
            data: kr_dps
        }
        data.push(kr_dps)

        // add the info about gpu2gpu transfers
        g2g_dps = []
        g2gs = profiling_data.device_logs[i].gpu2gpu_transfer_stats;
        for (let j = 0; j < g2gs.length; j++) {
            dp = {
                timeRange: [g2gs[j].start * 1e-5, g2gs[j].end * 1e-5],
                val: j
            }
            g2g_dps.push(dp);
        }
        g2g_dps = {
            label: gpu2gpu_copies,
            data: g2g_dps
        }
        data.push(g2g_dps)

        // add the info about cpu2gpu transfers
        c2g_dps = []
        c2gs = profiling_data.device_logs[i].cpu2gpu_transfer_stats;
        for (let j = 0; j < c2gs.length; j++) {
            cur_start_time = Math.min(cur_start_time, c2gs[j].start);
            dp = {
                timeRange: [c2gs[j].start * 1e-5, c2gs[j].end * 1e-5],
                val: j
            }
            c2g_dps.push(dp);
        }
        c2g_dps = {
            label: cpu2gpu_copies,
            data: c2g_dps
        }
        data.push(c2g_dps)

        evict_dps = []
        evict = profiling_data.device_logs[i].evicted_tensor_stats;
        for (let j = 0; j < evict.length; j++) {
            ev = {
                timeRange: [evict[j].start * 1e-5, evict[j].end * 1e-5],
                val: j
            }
            evict_dps.push(ev);
        }
        evict_dps = {
            label: evictions,
            data: evict_dps
        }
        data.push(evict_dps)


        free_dps = []
        free = profiling_data.device_logs[i].free_tensor_stats;
        for (let j = 0; j < free.length; j++) {
            fr = {
                timeRange: [free[j].start * 1e-5, free[j].start * 1e-5 + 20],
                val: j
            }
            free_dps.push(fr);
        }
        free_dps = {
            label: free_label,
            data: free_dps
        }
        data.push(free_dps)


        group = {
            data: data,
            group: "GPU " + i.toString()
        }
        groups.push(group)

        compute_time.push(cur_compute_time);
        start_times.push(cur_start_time);
        end_times.push(cur_end_time);
    }
    console.log(compute_time);
    console.log(start_times);
    console.log(end_times);

    var avg_percentage = 0;
    for(let i = 0; i < start_times.length; ++i) {
        avg_percentage += compute_time[i] / (end_times[i] - start_times[i]);
        console.log(compute_time[i] / (end_times[i] - start_times[i]));
    }
    console.log(avg_percentage / start_times.length);

    json_to_table = function (tableData) {

        var table = $('<table></table>');
        for (const [key, value] of Object.entries(tableData)) {
            var data = $('<tr></tr>');
            data.append($('<td>' + key + '</td>'));
            data.append($('<td>' + JSON.stringify(value) + '</td>'));
            table.append(data);
        }
        $("#dump").html(table);
    }

    TimelinesChart()(document.getElementById("timeline"))
        .xTickFormat(n => +n)
        .timeFormat('%Q')
        .zQualitative(true)
        .data(groups)
        .onSegmentClick(function (segment) {

            if (segment.label === gpu2gpu_copies) {
                device_idx = segment.group.split(" ")[1];
                t = profiling_data.device_logs[device_idx].gpu2gpu_transfer_stats[segment.val];
                json_to_table(t);
            }
            else if (segment.label === cpu2gpu_copies) {
                device_idx = segment.group.split(" ")[1];
                t = profiling_data.device_logs[device_idx].cpu2gpu_transfer_stats[segment.val];
                json_to_table(t);
            }
            else if (segment.label === kernels_run) {
                device_idx = segment.group.split(" ")[1];
                t = profiling_data.device_logs[device_idx].kernels_stats[segment.val];
                //t = profiling_data.device_logs[device_idx].kernels_scheduled[t.kernel_run_idx];
                json_to_table(t);
            }
            else if (segment.label === evictions) {
                device_idx = segment.group.split(" ")[1];
                t = profiling_data.device_logs[device_idx].evicted_tensor_stats[segment.val];
                json_to_table(t);
            }
            else if (segment.label === free_label) {
                device_idx = segment.group.split(" ")[1];
                t = profiling_data.device_logs[device_idx].free_tensor_stats[segment.val];
                json_to_table(t);
            }
        });
});
