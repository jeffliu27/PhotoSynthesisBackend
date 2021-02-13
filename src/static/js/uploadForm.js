$(document).ready(function(){
    $("#imageForm .basePhoto").change(function(e){
        if(window.FileReader){
            var reader = new FileReader();

            var files = $('#imageForm .uploadfile')[0].files;
            var basePhotoIndex = $("#imageForm .basePhoto")[0].value;

            var file = files[basePhotoIndex];

            reader.onload = function(e){
                $("#baseImageContainer").attr('src',e.target.result);
            }
            
            reader.readAsDataURL(file);
        } else {
            console.log("RIP")
        }
    });
    // send POST request
	$("#imageForm").submit(function(e) {
        e.preventDefault();
		
        var formData = new FormData();
        
        var fileCount = 0;
        var files = $('#imageForm .uploadfile')[0].files;
        var basePhotoIndex = $("#imageForm .basePhoto")[0].value;
        for (fileCount = 0; fileCount < files.length; fileCount++){
            formData.append('file[' + fileCount + ']', files[fileCount]);
        }
        formData.append("numFiles",fileCount);
        formData.append("basePhotoIndex",basePhotoIndex);

        var request = $.ajax({
            type: 'POST',
            url: '/api/generate_blink_free',
            data: formData,
            cache: false,
            processData: false,
            contentType: false
        });

        request.done(function(data,textStatus,jqXHR){
            console.log(jqXHR.status,jqXHR.responseJSON)

            var link = document.createElement('a');
            filePath = "/api/get_blink_free/" + data.session_id
            link. href = filePath;
            link.download = filePath.substr(filePath.lastIndexOf('/') + 1);
            link.click();
            // window.location.href = 
            // var getPhotoRequest = $.ajax({
            //     type: 'GET',
            //     url: '/api/get_blink_free/' + data.session_id,
            // });
        });

        request.fail(function(jqXHR,textStatus,errorThrown){
            console.log(jqXHR.status, jqXHR.responseJSON)
        });
    }); 
});
