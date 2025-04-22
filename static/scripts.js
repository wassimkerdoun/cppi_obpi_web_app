// Wait for the DOM to load
document.addEventListener('DOMContentLoaded', function () {
    // Get all navigation links
    const navLinks = document.querySelectorAll('.nav-link');

    // Load the active link from localStorage
    const activeLink = localStorage.getItem('activeLink');
    if (activeLink) {
        // Find the link with the matching `data-target` and add the active class
        const targetLink = document.querySelector(`.nav-link[data-target="${activeLink}"]`);
        if (targetLink) {
            targetLink.classList.add('active-link');
        }
    }

    // Add click event listener to each link
    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            // Prevent default behavior if you're handling navigation manually
            // e.preventDefault();

            // Remove the active class from all links
            navLinks.forEach(link => link.classList.remove('active-link'));

            // Add the active class to the clicked link
            this.classList.add('active-link');

            // Save the active link to localStorage
            const target = this.getAttribute('data-target');
            localStorage.setItem('activeLink', target);

            // Get the active page from the link's text content
            let pageName = this.textContent.toLowerCase();
            if (pageName.includes('computations')) {
                pageName = 'home';
            }
            localStorage.setItem('activePage', pageName);
        });
    });

    // Set initial active state based on stored page
    const currentPage = localStorage.getItem('activePage') || 'home';
    navLinks.forEach(link => {
        if (link.textContent.toLowerCase().includes(currentPage)) {
            link.classList.add('active-link');
        }
    });

    const dropdownButton = document.getElementById("method-dropdown");
    const dropdownItems = document.querySelectorAll(".dropdown-menu .dropdown-item");
    const selectedMethodInput = document.getElementById("selected-method");

    dropdownItems.forEach(item => {
        item.addEventListener("click", function (e) {
            e.preventDefault(); // Prevent default link behavior

            // Get the selected method's text and value
            const selectedText = item.textContent;
            const selectedValue = item.getAttribute("data-value");

            // Update the dropdown button text
            dropdownButton.textContent = selectedText;

            // Update the hidden input value
            selectedMethodInput.value = selectedValue;
        });
    });
});