$(document).ready(function() {
    $('#resumeForm').on('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // Display the results
                $('#predictedRole').text(response.predicted_role);
                $('#confidence').text((response.confidence * 100).toFixed(2) + '%');
                
                // Display skills
                const skillsContainer = $('#skills');
                skillsContainer.empty();
                response.extracted_skills.forEach(skill => {
                    skillsContainer.append(
                        $('<span>')
                            .addClass('skill-tag')
                            .text(skill)
                    );
                });
                
                // Show the result section
                $('#result').show();
            },
            error: function(xhr, status, error) {
                alert('Error: ' + (xhr.responseJSON?.error || 'Something went wrong'));
            }
        });
    });
}); 