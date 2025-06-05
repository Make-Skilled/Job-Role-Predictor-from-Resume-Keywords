document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.hash);
            if (target) {
                window.scrollTo({
                    top: target.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Navbar background change on scroll
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // Animate elements when they come into view
    const animateOnScroll = function() {
        const elements = document.querySelectorAll('.feature-card, .step');
        elements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementBottom = elementTop + element.offsetHeight;
            const viewportTop = window.scrollY;
            const viewportBottom = viewportTop + window.innerHeight;
            
            if (elementBottom > viewportTop && elementTop < viewportBottom) {
                element.classList.add('animate');
            }
        });
    };

    // Initial check for elements in view
    animateOnScroll();
    
    // Check for elements in view on scroll
    window.addEventListener('scroll', animateOnScroll);

    // File upload handling
    function updateFileName(input) {
        const fileInfo = document.getElementById('fileInfo');
        const uploadButton = document.getElementById('uploadButton');
        
        if (input.files && input.files[0]) {
            const file = input.files[0];
            const maxSize = 16 * 1024 * 1024; // 16MB
            
            // Check file size
            if (file.size > maxSize) {
                fileInfo.innerHTML = '<span class="text-danger">File is too large. Maximum size is 16MB.</span>';
                input.value = ''; // Clear the file input
                uploadButton.disabled = true;
                return;
            }
            
            // Update file info
            fileInfo.innerHTML = `
                <span class="text-success">
                    <i class="fas fa-check-circle"></i> Selected file: ${file.name}
                    (${(file.size / 1024 / 1024).toFixed(2)} MB)
                </span>
            `;
            uploadButton.disabled = false;
        } else {
            fileInfo.innerHTML = '';
            uploadButton.disabled = true;
        }
    }

    // Form submission handling
    document.getElementById('resumeForm').addEventListener('submit', function(e) {
        const uploadButton = document.getElementById('uploadButton');
        const originalText = uploadButton.innerHTML;
        
        // Show loading state
        uploadButton.disabled = true;
        uploadButton.innerHTML = `
            <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
            Analyzing...
        `;
    });

    // Flash message auto-dismiss
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
}); 